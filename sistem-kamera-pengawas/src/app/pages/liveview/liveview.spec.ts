import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Liveview } from './liveview';

describe('Liveview', () => {
  let component: Liveview;
  let fixture: ComponentFixture<Liveview>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Liveview]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Liveview);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
